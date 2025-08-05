"""Tests for ray_curator.backends.experimental.ray_data.utils module."""

import uuid
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest
import ray
from pytest import LogCaptureFixture

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.backends.experimental.ray_data.utils import (
    calculate_concurrency_for_actors_for_stage,
    execute_setup_on_node,
    get_available_cpu_gpu_resources,
)
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources

if TYPE_CHECKING:
    from ray_curator.tasks import Task


class TestGetAvailableCpuGpuResources:
    """Test class for utility functions in ray_data backend."""

    def test_get_available_cpu_gpu_resources_conftest(self, shared_ray_client: None):  # noqa: ARG002
        """Test get_available_cpu_gpu_resources function."""
        # Test with Ray resources from conftest.py
        cpus, gpus = get_available_cpu_gpu_resources()
        assert cpus == 11
        assert gpus == 0.0

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

    @patch("ray_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources")
    def test_calculate_concurrency_explicit_num_workers(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when num_workers is explicitly set."""
        mock_stage = Mock(num_workers=lambda: 4, resources=Resources(cpus=2.0, gpus=0.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == 4
        # Should not call get_resources if num_workers is set
        mock_get_resources.assert_not_called()

    @patch("ray_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 2.0))
    def test_calculate_concurrency_explicit_num_workers_zero_or_negative(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when num_workers is explicitly set to 0 or negative."""
        mock_stage = Mock(num_workers=lambda: 0, resources=Resources(cpus=2.0, gpus=0.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 4)
        mock_get_resources.assert_called_once()

    @patch("ray_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 2.0))
    def test_calculate_concurrency_cpu_only_constraint(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency with CPU-only constraint."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=2.0, gpus=0.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 4)
        mock_get_resources.assert_called_once()

    @patch("ray_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 4.0))
    def test_calculate_concurrency_gpu_only_constraint(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency with GPU-only constraint."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=0.0, gpus=1.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 4)
        mock_get_resources.assert_called_once()

    @patch("ray_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 4.0))
    def test_calculate_concurrency_both_cpu_gpu_constraints(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency with both CPU and GPU constraints."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=2.0, gpus=1.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 4)
        mock_get_resources.assert_called_once()

    @patch("ray_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(4.0, 8.0))
    def test_calculate_concurrency_cpu_more_limiting(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when CPU is more limiting than GPU."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=2.0, gpus=1.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 2)
        mock_get_resources.assert_called_once()

    @patch(
        "ray_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(16.0, 2.0)
    )
    def test_calculate_concurrency_gpu_more_limiting(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when GPU is more limiting than CPU."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=2.0, gpus=1.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 2)
        mock_get_resources.assert_called_once()

    @patch("ray_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 2.0))
    def test_calculate_concurrency_no_resource_requirements(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when stage has no resource requirements."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=0.0, gpus=0.0))
        with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
            calculate_concurrency_for_actors_for_stage(mock_stage)

        mock_get_resources.assert_called_once()

    @patch("ray_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(1.0, 0.0))
    def test_calculate_concurrency_insufficient_resources(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency when there are insufficient resources."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=4.0, gpus=2.0))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 0)
        mock_get_resources.assert_called_once()

    @patch("ray_curator.backends.experimental.ray_data.utils.get_available_cpu_gpu_resources", return_value=(8.0, 2.0))
    def test_calculate_concurrency_fractional_resources(self, mock_get_resources: MagicMock):
        """Test calculate_concurrency with fractional resource requirements."""
        mock_stage = Mock(num_workers=lambda: None, resources=Resources(cpus=0.5, gpus=0.25))
        assert calculate_concurrency_for_actors_for_stage(mock_stage) == (1, 8)
        mock_get_resources.assert_called_once()


class TestExecuteSetupOnNode:
    """Test class for execute_setup_on_node function."""

    def test_execute_setup_on_node_with_two_stages(
        self,
        shared_ray_client: None,  # noqa: ARG002
        tmp_path: Path,
        caplog: LogCaptureFixture,
    ):
        """Test execute_setup_on_node with two stages on the Ray cluster."""

        class MockStage1(ProcessingStage):
            _name = "mock_stage_1"
            _resources = Resources(cpus=1.0, gpus=0.0)

            def process(self, task: "Task") -> "Task":
                return task

            def setup_on_node(
                self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None
            ) -> None:
                # Write a file to record this call
                node_id = node_info.node_id if node_info else "unknown"
                worker_id = worker_metadata.worker_id if worker_metadata else "unknown"
                filename = f"{self.name}_{uuid.uuid4()}.txt"
                filepath = tmp_path / filename
                with open(filepath, "w") as f:
                    f.write(f"{node_id},{worker_id}\n")

        stage1 = MockStage1()
        stage2 = MockStage1().with_(name="mock_stage_2")

        # Test
        execute_setup_on_node([stage1, stage2])

        # Check the files written to the temp directory
        # Verify that NodeInfo and WorkerMetadata were passed correctly
        for stage_name in ["mock_stage_1", "mock_stage_2"]:
            stage_files = list(tmp_path.glob(f"{stage_name}_*.txt"))
            assert len(stage_files) == len(ray.nodes()), (
                f"Expected {len(ray.nodes())} calls to setup_on_node for {stage_name}, got {len(stage_files)}"
            )
            node_ids = set()
            for file_path in stage_files:
                content = file_path.read_text().strip()
                node_id, worker_id = content.split(",")
                assert worker_id == "", f"{stage_name} Worker ID should be empty string, got '{worker_id}'"
                node_ids.add(node_id)
            assert len(node_ids) == len(ray.nodes()), (
                f"Expected {len(ray.nodes())} different node IDs for {stage_name}, got {node_ids}"
            )
            assert node_ids == {node["NodeID"] for node in ray.nodes()}, (
                f"Expected node IDs to be the same as the Ray nodes, got {node_ids}"
            )

        # Check that there are exactly two log records that start with "Executing setup on node" and end with "for 2 stages"
        matching_logs = [
            record.message
            for record in caplog.records
            if record.message.startswith("Executing setup on node") and record.message.endswith("for 2 stages")
        ]
        # TODO: When we add a cluster then we should check the value of len(ray.nodes()) too
        assert len(matching_logs) == len(ray.nodes()), (
            f"Expected {len(ray.nodes())} logs for setup on node for 2 stages, got {len(matching_logs)}: {matching_logs}"
        )
