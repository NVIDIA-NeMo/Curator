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

from unittest import mock

import pytest

from nemo_curator.backends.ray_actor_pool.executor import RayActorPoolExecutor, _get_actor_options, _parse_runtime_env
from nemo_curator.backends.ray_actor_pool.utils import (
    calculate_optimal_actors_for_stage,
    calculate_optimal_actors_for_stage_with_wait,
    get_available_actor_pool_resources,
    update_resource_baseline,
)
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask


class TestRayActorPoolExecutor:
    def test_generate_task_batches_balances_weights(self) -> None:
        executor = RayActorPoolExecutor()
        tasks = [EmptyTask(dataset_name=str(i)) for i in range(6)]
        weights = {str(i): weight for i, weight in enumerate([10, 9, 8, 3, 2, 1])}

        weighted_batches = executor._generate_task_batches(
            tasks,
            num_output_tasks=2,
            task_weights=[10, 9, 8, 3, 2, 1],
        )
        unweighted_batches = executor._generate_task_batches(tasks, num_output_tasks=2)

        weighted_loads = [sum(weights[task.dataset_name] for task in batch) for batch in weighted_batches]
        unweighted_loads = [sum(weights[task.dataset_name] for task in batch) for batch in unweighted_batches]

        assert weighted_loads == [16, 17]
        assert unweighted_loads == [27, 6]
        assert max(weighted_loads) - min(weighted_loads) < max(unweighted_loads) - min(unweighted_loads)

    def test_generate_task_batches_distributes_zero_weights(self) -> None:
        executor = RayActorPoolExecutor()
        tasks = [EmptyTask(dataset_name=str(i)) for i in range(6)]

        batches = executor._generate_task_batches(tasks, num_output_tasks=3, task_weights=[0] * len(tasks))

        assert [len(batch) for batch in batches] == [2, 2, 2]

    @pytest.mark.parametrize(
        ("use_task_weights", "expected_weights"),
        [(None, [10, 1]), (True, [10, 1]), (False, None)],
    )
    def test_raft_stage_can_disable_task_weights(
        self, use_task_weights: bool | None, expected_weights: list[int] | None
    ) -> None:
        executor = RayActorPoolExecutor(show_progress=False)
        actor_pool = mock.Mock()
        actor_pool._idle_actors = [mock.Mock(), mock.Mock()]
        actor_pool.map_unordered.return_value = []
        stage = mock.Mock()
        stage.name = "raft_stage"
        stage_spec = {RayStageSpecKeys.IS_RAFT_ACTOR: True}
        if use_task_weights is not None:
            stage_spec[RayStageSpecKeys.USE_TASK_WEIGHTS] = use_task_weights
        stage.ray_stage_spec.return_value = stage_spec
        tasks = [EmptyTask(_metadata={"task_weight": weight}) for weight in [10, 1]]

        with (
            mock.patch("nemo_curator.backends.ray_actor_pool.executor.ray.get", return_value=None),
            mock.patch.object(executor, "_generate_task_batches", return_value=[]) as generate_batches,
        ):
            executor._process_stage_with_pool(actor_pool, stage, tasks)

        generate_batches.assert_called_once_with(tasks, num_output_tasks=2, task_weights=expected_weights)

    def test_task_weights_require_raft_stage(self) -> None:
        executor = RayActorPoolExecutor(show_progress=False)
        stage = mock.Mock()
        stage.name = "non_raft_stage"
        stage.ray_stage_spec.return_value = {RayStageSpecKeys.USE_TASK_WEIGHTS: False}

        with pytest.raises(RuntimeError, match="sets use_task_weights but is not a RAFT stage"):
            executor._process_stage_with_pool(mock.Mock(), stage, [])

    def test_parse_runtime_env(self):
        # With noset defined we should override it to be empty
        with_noset_defined = {"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": mock.ANY}}
        assert _parse_runtime_env(with_noset_defined) == {
            "env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": ""}
        }

        # we overwrite when config env_var is not provided
        without_env_var = {"some_other_key": "some_other_value"}
        assert _parse_runtime_env(without_env_var) == {
            "env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": ""},
            "some_other_key": "some_other_value",
        }

    @pytest.mark.parametrize(
        ("num_workers_per_node", "expected_strategy"),
        [(None, None), (2, "SPREAD")],
    )
    def test_actor_options_spread_per_node_workers(
        self, num_workers_per_node: float | None, expected_strategy: str | None
    ) -> None:
        stage = _stage_with_worker_sizing(
            num_workers=None,
            num_workers_per_node=num_workers_per_node,
            cpus=2.0,
            batch_size=1,
        )

        options = _get_actor_options(stage)

        assert options == {
            "num_cpus": 2.0,
            "num_gpus": 0.0,
            **({"scheduling_strategy": expected_strategy} if expected_strategy else {}),
        }

    @pytest.mark.parametrize(
        ("available_cpus", "expected_actors", "expected_warning"),
        [
            (8.0, 4, None),
            (2.0, 2, "requires 4 actors from num_workers(), but only 2 fit"),
        ],
    )
    def test_calculate_optimal_actors_respects_explicit_num_workers(
        self, available_cpus: float, expected_actors: int, expected_warning: str | None
    ) -> None:
        stage = _stage_with_worker_sizing(num_workers=4, num_workers_per_node=None, cpus=1.0, batch_size=10)

        with (
            mock.patch(
                "nemo_curator.backends.ray_actor_pool.utils.get_available_cpu_gpu_resources",
                return_value=(available_cpus, 0.0),
            ),
            mock.patch("nemo_curator.backends.ray_actor_pool.utils.logger.warning") as mock_warning,
        ):
            assert calculate_optimal_actors_for_stage(stage, num_tasks=1) == expected_actors

        if expected_warning is None:
            mock_warning.assert_not_called()
        else:
            mock_warning.assert_called_once()
            assert expected_warning in mock_warning.call_args.args[0]

    def test_wait_for_stage_resources_polls_cpu_and_gpu(self) -> None:
        stage = _stage_with_worker_sizing(num_workers=4, num_workers_per_node=None, cpus=1.0, batch_size=1)
        stage.resources.gpus = 1.0

        with (
            mock.patch(
                "nemo_curator.backends.ray_actor_pool.utils.get_available_cpu_gpu_resources",
                side_effect=[(0.0, 4.0), (2.0, 4.0), (4.0, 4.0)],
            ),
            mock.patch("nemo_curator.backends.ray_actor_pool.utils.time.sleep") as mock_sleep,
        ):
            assert calculate_optimal_actors_for_stage_with_wait(stage, 4, (4.0, 4.0), interval=0.2) == 4

        mock_sleep.assert_has_calls([mock.call(0.2), mock.call(0.2)])

    def test_wait_for_stage_resources_does_not_sleep_past_timeout(self) -> None:
        stage = _stage_with_worker_sizing(num_workers=4, num_workers_per_node=None, cpus=1.0, batch_size=1)

        with (
            mock.patch(
                "nemo_curator.backends.ray_actor_pool.utils.get_available_cpu_gpu_resources",
                return_value=(0.0, 0.0),
            ),
            mock.patch(
                "nemo_curator.backends.ray_actor_pool.utils.time.monotonic",
                side_effect=[0.0, 0.0, 0.0, 5.0],
            ),
            mock.patch("nemo_curator.backends.ray_actor_pool.utils.time.sleep") as mock_sleep,
            pytest.raises(TimeoutError),
        ):
            calculate_optimal_actors_for_stage_with_wait(stage, 4, (4.0, 0.0), timeout=5.0, interval=60.0)

        mock_sleep.assert_called_once_with(5.0)

    def test_wait_for_stage_resources_raises_when_only_partial_pool_fits_at_timeout(self) -> None:
        stage = _stage_with_worker_sizing(num_workers=4, num_workers_per_node=None, cpus=1.0, batch_size=1)
        stage.resources.gpus = 1.0

        with (
            mock.patch(
                "nemo_curator.backends.ray_actor_pool.utils.get_available_cpu_gpu_resources",
                return_value=(2.0, 1.0),
            ),
            pytest.raises(
                TimeoutError,
                match=(
                    r"intended 4-actor pool.*required CPUs=4\.0, GPUs=4\.0; "
                    r"available CPUs=2\.0, GPUs=1\.0"
                ),
            ),
        ):
            calculate_optimal_actors_for_stage_with_wait(stage, 4, (4.0, 4.0), timeout=0, interval=1)

    def test_wait_for_stage_resources_raises_when_no_actor_fits_at_timeout(self) -> None:
        stage = _stage_with_worker_sizing(num_workers=4, num_workers_per_node=None, cpus=1.0, batch_size=1)
        stage.resources.gpus = 1.0

        with (
            mock.patch(
                "nemo_curator.backends.ray_actor_pool.utils.get_available_cpu_gpu_resources",
                return_value=(0.0, 4.0),
            ),
            pytest.raises(TimeoutError, match=r"available CPUs=0\.0, GPUs=4\.0"),
        ):
            calculate_optimal_actors_for_stage_with_wait(stage, 4, (4.0, 4.0), timeout=0.5, interval=0.1)

    def test_wait_for_stage_resources_propagates_invalid_worker_count(self) -> None:
        stage = _stage_with_worker_sizing(
            num_workers=None,
            num_workers_per_node=1e308,
            cpus=1.0,
            batch_size=1,
        )

        with (
            mock.patch("nemo_curator.backends.ray_actor_pool.utils.get_alive_ray_node_count", return_value=2),
            pytest.raises(ValueError, match="too large"),
        ):
            calculate_optimal_actors_for_stage_with_wait(stage, 1, (0.0, 0.0))

    def test_actor_pool_resources_apply_reservations_and_update_baseline(self) -> None:
        with mock.patch(
            "nemo_curator.backends.ray_actor_pool.utils.get_available_cpu_gpu_resources",
            return_value=(128.0, 4.0),
        ):
            available = get_available_actor_pool_resources(reserved_cpus=2.0, reserved_gpus=1.0)
            baseline = update_resource_baseline((64.0, 1.0), reserved_cpus=2.0, reserved_gpus=1.0)

        assert available == (126.0, 3.0)
        assert baseline == (126.0, 3.0)

    def test_lsh_calculates_with_wait_for_each_band_iteration(self) -> None:
        executor = RayActorPoolExecutor()
        stage = mock.Mock()
        stage.name = "LSHStage"
        stage.resources = Resources(cpus=1.0, gpus=1.0)
        stage.actor_kwargs = {}
        stage.output_paths = ["first", "second"]
        stage.get_band_iterations.return_value = iter([(0, 1), (1, 2)])
        resource_baseline = (4.0, 4.0)

        with (
            mock.patch(
                "nemo_curator.backends.ray_actor_pool.executor.calculate_optimal_actors_for_stage_with_wait",
                return_value=4,
            ) as mock_calculate,
            mock.patch.object(
                executor, "_create_rapidsmpf_actors", side_effect=[[mock.sentinel.a], [mock.sentinel.b]]
            ),
            mock.patch.object(executor, "_process_shuffle_stage_with_rapidsmpf_actors", return_value=[]),
            mock.patch.object(executor, "_cleanup_actors"),
        ):
            executor._execute_lsh_stage(stage, [mock.sentinel.task], resource_baseline, 0.0, 0.0, 10.0, 0.5)

        assert mock_calculate.call_args_list == [
            mock.call(
                stage,
                1,
                resource_baseline,
                reserved_cpus=0.0,
                reserved_gpus=0.0,
                ignore_head_node=False,
                timeout=10.0,
                interval=0.5,
            ),
            mock.call(
                stage,
                1,
                resource_baseline,
                reserved_cpus=0.0,
                reserved_gpus=0.0,
                ignore_head_node=False,
                timeout=10.0,
                interval=0.5,
            ),
        ]

    @pytest.mark.parametrize(
        ("available_cpus", "expected_actors", "expected_warning"),
        [
            (8.0, 6, None),
            (4.0, 4, "requires 6 actors from num_workers_per_node()"),
        ],
    )
    def test_calculate_optimal_actors_respects_num_workers_per_node(
        self, available_cpus: float, expected_actors: int, expected_warning: str | None
    ) -> None:
        stage = _stage_with_worker_sizing(num_workers=None, num_workers_per_node=2, cpus=1.0, batch_size=10)

        with (
            mock.patch(
                "nemo_curator.backends.ray_actor_pool.utils.get_available_cpu_gpu_resources",
                return_value=(available_cpus, 0.0),
            ),
            mock.patch(
                "nemo_curator.backends.ray_actor_pool.utils.get_alive_ray_node_count", return_value=3
            ) as mock_node_count,
            mock.patch("nemo_curator.backends.ray_actor_pool.utils.logger.warning") as mock_warning,
        ):
            assert calculate_optimal_actors_for_stage(stage, num_tasks=1, ignore_head_node=True) == expected_actors

        mock_node_count.assert_called_once_with(ignore_head_node=True)
        if expected_warning is None:
            mock_warning.assert_not_called()
        else:
            mock_warning.assert_called_once()
            assert expected_warning in mock_warning.call_args.args[0]


def _stage_with_worker_sizing(
    *, num_workers: int | None, num_workers_per_node: float | None, cpus: float, batch_size: int
) -> mock.Mock:
    stage = mock.Mock()
    stage.name = "stage"
    stage.resources = Resources(cpus=cpus, gpus=0.0)
    stage.batch_size = batch_size
    stage.num_workers.return_value = num_workers
    stage.num_workers_per_node.return_value = num_workers_per_node
    return stage
