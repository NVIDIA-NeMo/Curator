"""Tests for ray_curator.backends.experimental.ray_data.utils module."""

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import ray
from pytest import LogCaptureFixture

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.backends.experimental.utils import execute_setup_on_node
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources

if TYPE_CHECKING:
    from ray_curator.tasks import Task


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
        stage2 = MockStage1().with_(name="mock_stage_2", resources=Resources(cpus=0.5, gpus=0.0))

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
