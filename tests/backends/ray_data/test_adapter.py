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

from unittest import mock

import pytest
from ray.data import ActorPoolStrategy, TaskPoolStrategy

from nemo_curator.backends.ray_data.adapter import RayDataStageAdapter
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage, Resources
from nemo_curator.tasks import EmptyTask


class RecordingDataset:
    """Minimal stand-in for Ray Data Dataset that records map_batches kwargs."""

    def __init__(self):
        self.map_batches_kwargs: dict[str, object] | None = None
        self.batch_size: int | None = None

    def map_batches(self, _fn: object, *, batch_size: int | None = None, **kwargs: object):
        self.batch_size = batch_size
        self.map_batches_kwargs = kwargs
        return self

    def repartition(self, **_: object):
        return self


class ConfigurableActorStage(ProcessingStage[EmptyTask, EmptyTask]):
    name = "configurable_actor"
    resources = Resources(cpus=2.0)
    batch_size = 7

    def __init__(self, ray_stage_spec: dict[str, object] | None = None, num_workers: int | None = None):
        self._ray_stage_spec = ray_stage_spec or {}
        self._num_workers = num_workers

    def ray_stage_spec(self) -> dict[str, object]:
        return {
            RayStageSpecKeys.IS_ACTOR_STAGE: True,
            **self._ray_stage_spec,
        }

    def num_workers(self) -> int | None:
        return self._num_workers

    def process(self, task: EmptyTask) -> EmptyTask:
        return task


class ConfigurableTaskStage(ConfigurableActorStage):
    name = "configurable_task"
    resources = Resources(cpus=2.0)
    batch_size = 7

    def ray_stage_spec(self) -> dict[str, object]:
        return self._ray_stage_spec


class ConflictingWorkerSizingTaskStage(ConfigurableTaskStage):
    def num_workers_per_node(self) -> int:
        return 2


class TestRayDataStageAdapter:
    def test_process_dataset_uses_compute_for_actor_stages_and_ray_default_for_task_stages(self):
        fixed_actor_kwargs = _map_batches_kwargs(ConfigurableActorStage(num_workers=3))
        autoscaling_actor_kwargs = _map_batches_kwargs(
            ConfigurableActorStage(
                ray_stage_spec={
                    RayStageSpecKeys.MIN_WORKERS: 2,
                    RayStageSpecKeys.MAX_WORKERS: 8,
                    RayStageSpecKeys.INITIAL_WORKERS: 4,
                }
            )
        )
        task_kwargs = _map_batches_kwargs(ConfigurableTaskStage())

        assert fixed_actor_kwargs["compute"] == ActorPoolStrategy(size=3)
        assert autoscaling_actor_kwargs["compute"] == ActorPoolStrategy(min_size=2, max_size=8, initial_size=4)
        assert "compute" not in task_kwargs
        for kwargs in (fixed_actor_kwargs, autoscaling_actor_kwargs, task_kwargs):
            assert "concurrency" not in kwargs

    def test_task_stage_uses_task_pool_strategy_for_num_workers(self):
        stage = ConfigurableTaskStage(num_workers=3)

        task_kwargs = _map_batches_kwargs(stage)

        assert task_kwargs["compute"] == TaskPoolStrategy(size=3)

    def test_task_stage_warns_on_actor_pool_sizing_keys(self):
        stage = ConfigurableTaskStage(
            ray_stage_spec={
                RayStageSpecKeys.MIN_WORKERS: 2,
                RayStageSpecKeys.MAX_WORKERS: 8,
                RayStageSpecKeys.INITIAL_WORKERS: 4,
            },
        )

        with mock.patch("nemo_curator.backends.ray_data.adapter.logger.warning") as mock_warning:
            task_kwargs = _map_batches_kwargs(stage)

        assert "compute" not in task_kwargs
        assert mock_warning.call_count == 1
        assert "Ignoring ray_stage_spec worker sizing keys" in mock_warning.call_args_list[0].args[0]

    def test_task_stage_uses_task_pool_strategy_for_num_workers_per_node(self):
        stage = ConfigurableTaskStage().with_(num_workers_per_node=2)

        with mock.patch("nemo_curator.backends.ray_data.adapter.get_alive_ray_node_count", return_value=3):
            task_kwargs = _map_batches_kwargs(stage)

        assert task_kwargs["compute"] == TaskPoolStrategy(size=6)
        assert task_kwargs["scheduling_strategy"] == "SPREAD"

    def test_actor_stage_uses_actor_pool_strategy_for_num_workers_per_node(self):
        stage = ConfigurableActorStage().with_(num_workers_per_node=2)

        with mock.patch("nemo_curator.backends.ray_data.adapter.get_alive_ray_node_count", return_value=3):
            actor_kwargs = _map_batches_kwargs(stage)

        assert actor_kwargs["compute"] == ActorPoolStrategy(size=6)
        assert actor_kwargs["scheduling_strategy"] == "SPREAD"

    def test_num_workers_per_node_passes_ignore_head_node_to_alive_node_count(self):
        stage = ConfigurableTaskStage().with_(num_workers_per_node=2)

        with mock.patch(
            "nemo_curator.backends.ray_data.adapter.get_alive_ray_node_count", return_value=2
        ) as mock_count:
            task_kwargs = _map_batches_kwargs(stage, ignore_head_node=True)

        mock_count.assert_called_once_with(ignore_head_node=True)
        assert task_kwargs["compute"] == TaskPoolStrategy(size=4)

    def test_num_workers_per_node_preserves_user_scheduling_strategy_override(self):
        stage = ConfigurableTaskStage().with_(
            ray_stage_spec={
                RayStageSpecKeys.RAY_REMOTE_ARGS: {"scheduling_strategy": "DEFAULT"},
            },
            num_workers_per_node=2,
        )

        with mock.patch("nemo_curator.backends.ray_data.adapter.get_alive_ray_node_count", return_value=3):
            task_kwargs = _map_batches_kwargs(stage)

        assert task_kwargs["compute"] == TaskPoolStrategy(size=6)
        assert task_kwargs["scheduling_strategy"] == "DEFAULT"

    @pytest.mark.parametrize("num_workers", [0, 3])
    def test_num_workers_per_node_rejects_explicit_stage_num_workers(self, num_workers: int):
        # Mutual exclusion is enforced at stage construction, not in the adapter.
        with pytest.raises(ValueError, match=r"num_workers.*num_workers_per_node"):
            ConflictingWorkerSizingTaskStage(num_workers=num_workers)

    def test_source_fanout_task_stage_uses_task_pool_strategy_for_single_worker_default(self):
        stage = ConfigurableTaskStage(
            ray_stage_spec={RayStageSpecKeys.IS_FANOUT_STAGE: True},
            num_workers=1,
        )

        with mock.patch("nemo_curator.backends.ray_data.adapter.logger.warning") as mock_warning:
            task_kwargs = _map_batches_kwargs(stage)

        assert task_kwargs["compute"] == TaskPoolStrategy(size=1)
        mock_warning.assert_not_called()

    def test_process_dataset_rejects_managed_ray_remote_args(self):
        stage = ConfigurableActorStage(
            ray_stage_spec={
                RayStageSpecKeys.RAY_REMOTE_ARGS: {"compute": ActorPoolStrategy(size=2)},
            }
        )

        with pytest.raises(ValueError, match="must not override Curator-managed map_batches arguments"):
            _map_batches_kwargs(stage)

    def test_build_resource_kwargs_uses_ray_num_cpus_from_spec_over_resources_cpus(self):
        stage = ConfigurableActorStage(ray_stage_spec={RayStageSpecKeys.RAY_NUM_CPUS: 1.0})
        kwargs = _map_batches_kwargs(stage)
        assert kwargs["num_cpus"] == 1.0

    def test_build_resource_kwargs_falls_back_to_resources_cpus_when_ray_num_cpus_absent(self):
        stage = ConfigurableActorStage()
        kwargs = _map_batches_kwargs(stage)
        assert kwargs["num_cpus"] == stage.resources.cpus


def _map_batches_kwargs(stage: ProcessingStage, ignore_head_node: bool = False) -> dict[str, object]:
    dataset = RecordingDataset()
    RayDataStageAdapter(stage, ignore_head_node=ignore_head_node).process_dataset(dataset)  # type: ignore[arg-type]
    assert dataset.map_batches_kwargs is not None
    assert dataset.batch_size == stage.batch_size
    return dataset.map_batches_kwargs
