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

from unittest.mock import MagicMock, Mock, patch

import pytest
from ray.data import ActorPoolStrategy

from nemo_curator.backends.ray_data.utils import (
    get_actor_compute_strategy_for_stage,
)
from nemo_curator.backends.utils import RayStageSpecKeys, get_available_cpu_gpu_resources
from tests.backends.test_utils import reset_head_node_cache  # noqa: F401


class TestGetAvailableCpuGpuResources:
    # TODO: Move this to tests/backends/test_utils.py
    """Test class for utility functions in ray_data backend."""

    def test_get_available_cpu_gpu_resources_conftest(self, shared_ray_client: None):
        """Test get_available_cpu_gpu_resources function."""
        # Test with Ray resources from conftest.py
        cpus, gpus = get_available_cpu_gpu_resources()
        assert cpus == 11
        # GPU count depends on whether GPU tests are running in this session
        # Can be 0 (CPU-only) or 2 (GPU-enabled) depending on test selection
        assert gpus in [0.0, 2.0]

    @pytest.mark.usefixtures("reset_head_node_cache")
    def test_get_resources_with_ignore_head_node(
        self,
        shared_ray_client: None,
    ):
        """Test get_available_cpu_gpu_resources with ignore_head_node=True to skip head node.
        Since this test is run with the head node, the resources should be 0."""
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
    ):
        mock_stage = Mock(num_workers=lambda: num_workers, ray_stage_spec=lambda: ray_stage_spec)
        mock_stage.name = "stage"

        with patch("nemo_curator.backends.ray_data.utils.logger.warning") as mock_warning:
            assert get_actor_compute_strategy_for_stage(mock_stage) == expected

        if expected_warning is None:
            mock_warning.assert_not_called()
        else:
            mock_warning.assert_called_once()
            assert expected_warning in mock_warning.call_args.args[0]

    def test_actor_compute_strategy_rejects_invalid_sizing(self):
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
