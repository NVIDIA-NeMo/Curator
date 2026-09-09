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

from nemo_curator.backends.ray_actor_pool.executor import RayActorPoolExecutor, _parse_runtime_env
from nemo_curator.backends.ray_actor_pool.utils import (
    calculate_optimal_actors_for_stage,
    calculate_optimal_actors_for_stage_with_wait,
    get_available_actor_pool_resources,
    update_resource_baseline,
)
from nemo_curator.stages.resources import Resources


class TestRayActorPoolExecutor:
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
        ("available_cpus", "expected_actors", "expected_warning"),
        [
            (8.0, 4, None),
            (2.0, 2, "requires 4 actors from num_workers(), but only 2 fit"),
        ],
    )
    def test_calculate_optimal_actors_respects_explicit_num_workers(
        self, available_cpus: float, expected_actors: int, expected_warning: str | None
    ) -> None:
        stage = _stage_with_num_workers(num_workers=4, cpus=1.0, batch_size=10)

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
        stage = _stage_with_num_workers(num_workers=4, cpus=1.0, batch_size=1)
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
        stage = _stage_with_num_workers(num_workers=4, cpus=1.0, batch_size=1)

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
        stage = _stage_with_num_workers(num_workers=4, cpus=1.0, batch_size=1)
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
        stage = _stage_with_num_workers(num_workers=4, cpus=1.0, batch_size=1)
        stage.resources.gpus = 1.0

        with (
            mock.patch(
                "nemo_curator.backends.ray_actor_pool.utils.get_available_cpu_gpu_resources",
                return_value=(0.0, 4.0),
            ),
            pytest.raises(TimeoutError, match=r"available CPUs=0\.0, GPUs=4\.0"),
        ):
            calculate_optimal_actors_for_stage_with_wait(stage, 4, (4.0, 4.0), timeout=0.5, interval=0.1)

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


def _stage_with_num_workers(*, num_workers: int, cpus: float, batch_size: int) -> mock.Mock:
    stage = mock.Mock()
    stage.name = "stage"
    stage.resources = Resources(cpus=cpus, gpus=0.0)
    stage.batch_size = batch_size
    stage.num_workers.return_value = num_workers
    return stage
