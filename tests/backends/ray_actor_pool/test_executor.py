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

from nemo_curator.backends.ray_actor_pool.executor import _parse_runtime_env
from nemo_curator.backends.ray_actor_pool.utils import calculate_optimal_actors_for_stage
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


def _stage_with_num_workers(*, num_workers: int, cpus: float, batch_size: int) -> mock.Mock:
    stage = mock.Mock()
    stage.name = "stage"
    stage.resources = Resources(cpus=cpus, gpus=0.0)
    stage.batch_size = batch_size
    stage.num_workers.return_value = num_workers
    return stage
