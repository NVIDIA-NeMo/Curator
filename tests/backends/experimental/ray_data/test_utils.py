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

from nemo_curator.backends.experimental.ray_data.utils import (
    calculate_concurrency_for_actors_for_stage,
    get_available_cpu_gpu_resources,
)
from nemo_curator.stages.resources import Resources


class TestGetAvailableCpuGpuResources:
    """Test class for utility functions in ray_data backend."""

    def test_get_available_cpu_gpu_resources_conftest(self, shared_ray_client: None):  # noqa: ARG002
        """Test get_available_cpu_gpu_resources function."""
        # Test with Ray resources from conftest.py
        cpus, gpus = get_available_cpu_gpu_resources()
        assert cpus == 11
        # GPU count depends on whether GPU tests are running in this session
        # Can be 0 (CPU-only) or 2 (GPU-enabled) depending on test selection
        assert gpus in [0.0, 2.0]

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

    @patch("nemo_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources")
    def test_calculate_concurrency_explicit_num_workers(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when num_workers is explicitly set."""
        mock_stage = Mock(num_workers=lambda: 4, resources=Resources(cpus=2.0, gpus=0.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == 4
        # Should not call get_resources if num_workers is set
        mock_get_resources.assert_not_called()

    @patch("nemo_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 2.0))
    def test_calculate_concurrency_explicit_num_workers_zero_or_negative(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when num_workers is explicitly set to 0 or negative."""
        mock_stage = Mock(num_workers=lambda: 0, resources=Resources(cpus=2.0, gpus=0.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 4)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 2.0))
    def test_calculate_concurrency_cpu_only_constraint(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency with CPU-only constraint."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=2.0, gpus=0.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 4)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 4.0))
    def test_calculate_concurrency_gpu_only_constraint(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency with GPU-only constraint."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=0.0, gpus=1.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 4)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 4.0))
    def test_calculate_concurrency_both_cpu_gpu_constraints(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency with both CPU and GPU constraints."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=2.0, gpus=1.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 4)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(4.0, 8.0))
    def test_calculate_concurrency_cpu_more_limiting(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when CPU is more limiting than GPU."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=2.0, gpus=1.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 2)
        mock_get_resources.assert_called_once()

    @patch(
        "nemo_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(16.0, 2.0)
    )
    def test_calculate_concurrency_gpu_more_limiting(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when GPU is more limiting than CPU."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=2.0, gpus=1.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 2)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 2.0))
    def test_calculate_concurrency_no_resource_requirements(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when stage has no resource requirements."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=0.0, gpus=0.0))
        with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
            calculate_concurrency_for_actors_for_stage(mock_stage)

        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(1.0, 0.0))
    def test_calculate_concurrency_insufficient_resources(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when there are insufficient resources."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=4.0, gpus=2.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 0)
        mock_get_resources.assert_called_once()

    @patch("nemo_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 2.0))
    def test_calculate_concurrency_fractional_resources(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency with fractional resource requirements."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=0.5, gpus=0.25))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 8)
        mock_get_resources.assert_called_once()
